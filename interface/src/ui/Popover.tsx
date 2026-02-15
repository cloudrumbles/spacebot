"use client";

import * as React from "react";
import * as PopoverPrimitive from "@radix-ui/react-popover";
import { cva, type VariantProps } from "class-variance-authority";
import { cx } from "./utils";

const popoverContentStyles = cva(
  [
    "z-50 w-72 rounded-lg border border-app-line bg-app-box p-4 shadow-lg",
    "outline-none data-[state=open]:animate-in data-[state=closed]:animate-out",
    "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
    "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
    "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
    "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
  ]
);

export interface PopoverContentProps
  extends React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>,
    VariantProps<typeof popoverContentStyles> {}

export const Popover = PopoverPrimitive.Root;

export const PopoverTrigger = PopoverPrimitive.Trigger;

export const PopoverAnchor = PopoverPrimitive.Anchor;

export const PopoverContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  PopoverContentProps
>(({ className, align = "center", sideOffset = 4, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={cx(popoverContentStyles(), className)}
      {...props}
    />
  </PopoverPrimitive.Portal>
));

PopoverContent.displayName = PopoverPrimitive.Content.displayName;

export const PopoverClose = PopoverPrimitive.Close;

export interface PopoverHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
}

export const PopoverHeader: React.FC<PopoverHeaderProps> = ({
  title,
  description,
  className,
  children,
  ...props
}) => (
  <div className={cx("space-y-1.5", className)} {...props}>
    {title && <h4 className="text-sm font-semibold text-ink">{title}</h4>}
    {description && <p className="text-xs text-ink-dull">{description}</p>}
    {children}
  </div>
);

export interface PopoverFooterProps extends React.HTMLAttributes<HTMLDivElement> {}

export const PopoverFooter: React.FC<PopoverFooterProps> = ({
  className,
  children,
  ...props
}) => (
  <div
    className={cx(
      "mt-4 flex items-center justify-end gap-2",
      className
    )}
    {...props}
  >
    {children}
  </div>
);
