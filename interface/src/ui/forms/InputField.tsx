"use client";

import * as React from "react";
import { useFormContext, Controller } from "react-hook-form";
import { Input, type InputProps, Label } from "../Input";
import { ErrorMessage } from "./Form";
import { cx } from "../utils";

export interface InputFieldProps extends Omit<InputProps, "name" | "error"> {
  name: string;
  label?: string;
  description?: string;
  className?: string;
}

export const InputField = React.forwardRef<HTMLInputElement, InputFieldProps>(
  ({ name, label, description, className, ...props }, ref) => {
    const { control, formState } = useFormContext();
    const error = formState.errors[name]?.message as string | undefined;

    return (
      <Controller
        name={name}
        control={control}
        render={({ field }) => (
          <div className={cx("space-y-1.5", className)}>
            {label && <Label htmlFor={name}>{label}</Label>}
            <Input
              {...field}
              {...props}
              ref={ref}
              id={name}
              error={!!error}
            />
            {description && !error && (
              <p className="text-xs text-ink-dull">{description}</p>
            )}
            {error && <ErrorMessage error={error} />}
          </div>
        )}
      />
    );
  }
);

InputField.displayName = "InputField";
